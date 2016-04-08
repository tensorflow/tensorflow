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

#ifndef TENSORFLOW_FRAMEWORK_LOOKUP_INTERFACE_H_
#define TENSORFLOW_FRAMEWORK_LOOKUP_INTERFACE_H_

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
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
  virtual Status Find(const Tensor& keys, Tensor* values,
                      const Tensor& default_value) = 0;

  // Returns the number of elements in the table.
  virtual size_t size() const = 0;

  // Returns the data type of the key.
  virtual DataType key_dtype() const = 0;

  // Returns the data type of the value.
  virtual DataType value_dtype() const = 0;

  string DebugString() override { return "A lookup table"; }

  // Returns an InitializableLookupTable, a subclass of LookupInterface, if the
  // current object is an InitializableLookupTable. Otherwise, returns nullptr.
  virtual InitializableLookupTable* GetInitializableLookupTable() {
    return nullptr;
  }

 protected:
  virtual ~LookupInterface() = default;

  // Check format of the key and value tensors.
  // Returns OK if all the following requirements are satisfied, otherwise it
  // returns InvalidArgument:
  // - DataType of the tensor key equals to the table key_dtype
  // - DataType of the test value equals to the table value_dtype
  // - key and value have the same size and shape
  Status CheckKeyAndValueTensors(const Tensor& keys, const Tensor& values);

  // Check the arguments of a find operation. Returns OK if all the following
  // requirements are satisfied, otherwise it returns InvalidArgument:
  // - All requirements of CheckKeyAndValueTensors
  // - default_value type equals to the table value_dtype
  // - default_value is scalar
  Status CheckFindArguments(const Tensor& keys, const Tensor& values,
                            const Tensor& default_value);
};

}  // namespace lookup
}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_LOOKUP_INTERFACE_H_
