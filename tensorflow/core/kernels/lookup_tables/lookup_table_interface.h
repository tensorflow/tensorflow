/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_LOOKUP_TABLES_LOOKUP_TABLE_INTERFACE_H_
#define TENSORFLOW_CORE_KERNELS_LOOKUP_TABLES_LOOKUP_TABLE_INTERFACE_H_

#include <cstddef>
#include <string>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace tables {

// Interface for key-value pair lookups with support for heterogeneous keys.
// This class contains two main kinds of methods: methods which operate on
// a batch of inputs and methods which do not. The latter have the prefix
// 'Unsafe'. Clients must call the corresponding status methods to determine
// whether they are safe to call within a code block.
// Implementations must guarantee thread-safety when GetMutex is used to
// synchronize method access.
template <typename HeterogeneousKeyType, typename ValueType>
class LookupTableInterface : public ResourceBase {
 public:
  using heterogeneous_key_type = HeterogeneousKeyType;
  using value_type = ValueType;
  using key_type = heterogeneous_key_type;

  // Return value should be used to synchronize read/write access to
  // all public methods. If null, no synchronization is needed.
  virtual mutex* GetMutex() const = 0;

  // Insert the KV pair into the underlying table. If a key equivalent to key
  // already exists in the underlying table, its corresponding value is
  // overridden. Returns true only if the key was inserted for the first time.
  // Undefined if TableUnbatchedInsertStatus() != OK.
  virtual bool UnsafeInsertOrAssign(const HeterogeneousKeyType& key,
                                    const ValueType& value) = 0;

  // Returns OK if it is safe to call InsertOrAssign.
  // Once OK is returned, it is safe to call InsertOrAssign for the rest of the
  // program.
  virtual Status TableUnbatchedInsertStatus() const TF_MUST_USE_RESULT = 0;

  // Stores each KV pair {keys[i], values[i]} in the underlying map, overriding
  // pre-existing pairs which have equivalent keys.
  // keys and values should have the same size.
  virtual Status BatchInsertOrAssign(
      absl::Span<const HeterogeneousKeyType> keys,
      absl::Span<const ValueType> values) = 0;

  // Prefetch key_to_find into implementation defined data caches.
  // Implementations are free to leave this a no-op.
  // Undefined if TableUnbatchedLookupStatus() != OK.
  virtual void UnsafePrefetchKey(
      const HeterogeneousKeyType& key_to_find) const {}

  // Returns true if and only if the table contains key_to_find.
  // Undefined if TableUnbatchedLookupStatus() != OK.
  virtual bool UnsafeContainsKey(
      const HeterogeneousKeyType& key_to_find) const = 0;

  // Lookup the value for key_to_find. This value must always be well-defined,
  // even when ContainsKey(key_to_find) == false. When
  // dv = DefaultValue() != absl::nullopt and ContainsKey(key_to_find) == false,
  // dv is returned.
  // Undefined if TableUnbatchedLookupStatus() != OK.
  virtual ValueType UnsafeLookupKey(
      const HeterogeneousKeyType& key_to_find) const = 0;

  // Returns OK if it is safe to call PrefetchKey, ContainsKey, and
  // UnsafeLookupKey.
  // If OK is returned, it is safe to call these methods until the next
  // non-const method of this class is called.
  virtual Status TableUnbatchedLookupStatus() const TF_MUST_USE_RESULT = 0;

  // Lookup the values for keys and store them in values.
  // prefetch_lookahead is used to prefetch the key at index
  // i + prefetch_lookahead at the ith iteration of the implemented loop.
  // keys and values must have the same size.
  virtual Status BatchLookup(absl::Span<const HeterogeneousKeyType> keys,
                             absl::Span<ValueType> values,
                             int64 prefetch_lookahead) const = 0;

  // Returns the number of elements in the table.
  // Undefined if SizeStatus() != OK.
  virtual size_t UnsafeSize() const = 0;

  // Returns OK if the return value of UnsafeSize() is always well-defined.
  virtual Status SizeStatus() const TF_MUST_USE_RESULT = 0;

  // If non-null value is returned, LookupKey returns that value only for keys
  // which satisfy ContainsKey(key_to_find) == false.
  virtual const absl::optional<const ValueType> DefaultValue() const = 0;

  string DebugString() const override { return "A lookup table"; }

  ~LookupTableInterface() override = default;
};

}  // namespace tables
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LOOKUP_TABLES_LOOKUP_TABLE_INTERFACE_H_
