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

#include "absl/types/span.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace tables {

// Interface for key-value pair lookups with support for heterogeneous keys.
// One of the key high performance abstractions introduced here is the
// hierarchy of a primary and secondary (or fallback) table.
// The primary table can be considered the main data structure into which
// key-value pairs may be inserted.
// When a key is requested, LookupTableInterface::LookupKey semantics expect
// that the primary table be queried first. The secondary table is queried
// when the primary table lookup fails (eg. does not contain a requested key).
// Lookups in the secondary table must always return well defined values in
// this case.
// LookupTableInterface features batch and serial table insert/lookup
// methods. Serial methods have corresponding status methods which say
// whether the table supports serial inserts/lookups which are guaranteed to
// succeed.
// For example, an implementation may only support a one-time batch population
// after which further insertions are undefined. Another may not permit lookups
// before its size is non zero.
// In each of these cases, the success or failure of inserts/lookups is not
// dependent on the parameter values. On the other hand, once the table is in
// ready state, LookupTableInterface semantics guarantee that all lookups and
// inserts will succeed.
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
  // Undefined if TableInsertStatus() != OK.
  virtual bool InsertOrAssign(const HeterogeneousKeyType& key,
                              const ValueType& value) = 0;

  // Returns OK if it is safe to call InsertOrAssign.
  // Once OK is returned, it is safe to call InsertOrAssign for the rest of the
  // program.
  virtual Status TableInsertStatus() const TF_MUST_USE_RESULT = 0;

  // Stores each KV pair {keys[i], values[i]} in the underlying map, overriding
  // pre-existing pairs with equivalent keys.
  // keys and values should have the same size.
  virtual Status BatchInsertOrAssign(
      absl::Span<const HeterogeneousKeyType> keys,
      absl::Span<const ValueType> values) = 0;

  // Prefetch key_to_find into implementation defined data caches.
  // Implementations are free to leave this a no-op.
  // Undefined if TableLookupStatus() != OK.
  virtual void PrefetchKey(const HeterogeneousKeyType& key_to_find) const {}

  // Returns true if and only if the primary table contains key_to_find.
  // What constitutes the primary table is implementation defined.
  // Undefined if TableLookupStatus() != OK.
  virtual bool ContainsKey(const HeterogeneousKeyType& key_to_find) const = 0;

  // Lookup the value for key_to_find in the primary. If this lookup fails
  // (for eg. the key does not exist), the value in the secondary table is
  // returned.
  // Undefined if TableLookupStatus() != OK.
  virtual ValueType LookupKey(
      const HeterogeneousKeyType& key_to_find) const = 0;

  // Lookup the value for key_to_find in the primary table and return it.
  // If the lookup fails, the return value is undefined unless
  // PrimaryTableDefaultValue() != NULL in which case
  // *PrimaryTableDefaultValue() is returned.
  // Undefined if TableLookupStatus() != OK.
  virtual ValueType LookupKeyInPrimaryTable(
      const HeterogeneousKeyType& key_to_find) const = 0;

  // Lookup the value for key_to_find in the secondary table and return it.
  // This value may not be well defined if key_to_find is in the primary table.
  // Undefined if TableLookupStatus() != OK.
  virtual ValueType LookupKeyInSecondaryTable(
      const HeterogeneousKeyType& key_to_find) const = 0;

  // Lookup the values for keys in the primary or secondary table and store
  // them in values. prefetch_lookahead is used to prefetch the key at index
  // i + prefetch_lookahead at the ith iteration of the implemented loop.
  // Undefined if TableLookupStatus() != OK.
  virtual Status BatchLookup(absl::Span<const HeterogeneousKeyType> keys,
                             absl::Span<ValueType> values,
                             int64 prefetch_lookahead) const = 0;

  // Returns OK if it is safe to call PrefetchKey, ContainsKey, LookupKey,
  // LookupKeyInPrimaryTable, and LookupKeyInSecondaryTable.
  // Once OK is returned, it is safe to call these methods until the next
  // non-const method of this class is called.
  virtual Status TableLookupStatus() const TF_MUST_USE_RESULT = 0;

  // If non-null value is returned, *value is guaranteed to not be a value in
  // the primary table. LookupKeyInPrimaryTable returns *value for keys
  // which are not in the primary table.
  virtual const ValueType* PrimaryTableDefaultValue() const = 0;

  // Returns the number of elements in the primary table.
  // Undefined if SizeStatus() != OK.
  virtual size_t size() const = 0;

  // Returns OK if the return value of size() is well defined.
  virtual Status SizeStatus() const TF_MUST_USE_RESULT = 0;

  string DebugString() const override {
    return strings::StrCat("A lookup table of size: ", size());
  }

  ~LookupTableInterface() override = default;
};

}  // namespace tables
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LOOKUP_TABLES_LOOKUP_TABLE_INTERFACE_H_
