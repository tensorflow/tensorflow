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

#ifndef TENSORFLOW_CORE_FRAMEWORK_LOOKUP_INTERFACE_H_
#define TENSORFLOW_CORE_FRAMEWORK_LOOKUP_INTERFACE_H_

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class OpKernelContext;

namespace lookup {

// Forward declaration so we can define GetInitializableLookupTable() in
// LookupInterface.
class InitializableLookupTable;

// Lookup interface for batch lookups used by table lookup ops.
class LookupInterface : public ResourceBase {
 public:
  // Performs batch lookups, for every element in the key tensor, Find returns
  // the corresponding value into the values tensor.
  // If an element is not present in the table, the given default value is used.

  // For tables that require initialization, Find is available once the table
  // is marked as initialized.

  // Returns the following statuses:
  // - OK: when the find finishes successfully.
  // - FailedPrecondition: if the table is not initialized.
  // - InvalidArgument: if any of the preconditions on the lookup key or value
  //   fails.
  // - In addition, other implementations may provide another non-OK status
  //   specific to their failure modes.
  virtual absl::Status Find(OpKernelContext* ctx, const Tensor& keys,
                            Tensor* values, const Tensor& default_value) = 0;

  // Inserts elements into the table. Each element of the key tensor is
  // associated with the corresponding element in the value tensor.
  // This method is only implemented in mutable tables that can be updated over
  // the execution of the graph. It returns Status::NotImplemented for read-only
  // tables that are initialized once before they can be looked up.

  // Returns the following statuses:
  // - OK: when the insert finishes successfully.
  // - InvalidArgument: if any of the preconditions on the lookup key or value
  //   fails.
  // - Unimplemented: if the table does not support insertions.
  virtual absl::Status Insert(OpKernelContext* ctx, const Tensor& keys,
                              const Tensor& values) = 0;

  // Removes elements from the table.
  // This method is only implemented in mutable tables that can be updated over
  // the execution of the graph. It returns Status::NotImplemented for read-only
  // tables that are initialized once before they can be looked up.

  // Returns the following statuses:
  // - OK: when the remove finishes successfully.
  // - InvalidArgument: if any of the preconditions on the lookup key fails.
  // - Unimplemented: if the table does not support removals.
  virtual absl::Status Remove(OpKernelContext* ctx, const Tensor& keys) = 0;

  // Returns the number of elements in the table.
  virtual size_t size() const = 0;

  // Exports the values of the table to two tensors named keys and values.
  // Note that the shape of the tensors is completely up to the implementation
  // of the table and can be different than the tensors used for the Insert
  // function above.
  virtual absl::Status ExportValues(OpKernelContext* ctx) = 0;

  // Imports previously exported keys and values.
  // As mentioned above, the shape of the keys and values tensors are determined
  // by the ExportValues function above and can be different than for the
  // Insert function.
  virtual absl::Status ImportValues(OpKernelContext* ctx, const Tensor& keys,
                                    const Tensor& values) = 0;

  // Returns the data type of the key.
  virtual DataType key_dtype() const = 0;

  // Returns the data type of the value.
  virtual DataType value_dtype() const = 0;

  // Returns the shape of a key in the table.
  virtual TensorShape key_shape() const = 0;

  // Returns the shape of a value in the table.
  virtual TensorShape value_shape() const = 0;

  // Check format of the key and value tensors for the Insert function.
  // Returns OK if all the following requirements are satisfied, otherwise it
  // returns InvalidArgument:
  // - DataType of the tensor keys equals to the table key_dtype
  // - DataType of the tensor values equals to the table value_dtype
  // - the values tensor has the required shape given keys and the tables's
  //   value shape.
  virtual absl::Status CheckKeyAndValueTensorsForInsert(const Tensor& keys,
                                                        const Tensor& values);

  // Similar to the function above but instead checks eligibility for the Import
  // function.
  virtual absl::Status CheckKeyAndValueTensorsForImport(const Tensor& keys,
                                                        const Tensor& values);

  // Check format of the key tensor for the Remove function.
  // Returns OK if all the following requirements are satisfied, otherwise it
  // returns InvalidArgument:
  // - DataType of the tensor keys equals to the table key_dtype
  virtual absl::Status CheckKeyTensorForRemove(const Tensor& keys);

  // Check the arguments of a find operation. Returns OK if all the following
  // requirements are satisfied, otherwise it returns InvalidArgument:
  // - DataType of the tensor keys equals to the table key_dtype
  // - DataType of the tensor default_value equals to the table value_dtype
  // - the default_value tensor has the required shape given keys and the
  //   tables's value shape.
  absl::Status CheckFindArguments(const Tensor& keys,
                                  const Tensor& default_value);

  string DebugString() const override {
    return strings::StrCat("A lookup table of size: ", size());
  }

  // Returns an InitializableLookupTable, a subclass of LookupInterface, if the
  // current object is an InitializableLookupTable. Otherwise, returns nullptr.
  virtual InitializableLookupTable* GetInitializableLookupTable() {
    return nullptr;
  }

 protected:
  ~LookupInterface() override = default;

  // Makes sure that the key and value tensor DataType's match the table
  // key_dtype and value_dtype.
  absl::Status CheckKeyAndValueTypes(const Tensor& keys, const Tensor& values);

  // Makes sure that the provided shape is consistent with the table keys shape.
  absl::Status CheckKeyShape(const TensorShape& shape);

 private:
  absl::Status CheckKeyAndValueTensorsHelper(const Tensor& keys,
                                             const Tensor& values);
};

}  // namespace lookup
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_LOOKUP_INTERFACE_H_
