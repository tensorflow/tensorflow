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

#ifndef TENSORFLOW_CORE_KERNELS_LOOKUP_TABLES_RESOURCE_INTERFACE_TEMPLATES_H_
#define TENSORFLOW_CORE_KERNELS_LOOKUP_TABLES_RESOURCE_INTERFACE_TEMPLATES_H_

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace tables {

// Interface for resources with mutable state.
class SynchronizedInterface : public virtual ResourceBase {
 public:
  // Return value should be used to synchronize read/write access to
  // all public methods. If null, no synchronization is needed.
  virtual mutex* GetMutex() const = 0;
};

// Interface for containers which support batch lookups.
template <typename ValueType, typename... KeyContext>
class InsertOrAssignInterface : public virtual SynchronizedInterface {
 public:
  using value_type = ValueType;

  // Stores each KV pair {keys[i], values[i]} in the underlying map, overriding
  // pre-existing pairs which have equivalent keys.
  // keys and values should have the same size.
  virtual Status InsertOrAssign(KeyContext... key_context,
                                ValueType values) = 0;
};

// Interface for containers which support lookups.
template <typename ValueType, typename... KeyContext>
class LookupInterface : public virtual SynchronizedInterface {
 public:
  using value_type = ValueType;

  // Lookup the values for keys and store them in values.
  // prefetch_lookahead is used to prefetch the key at index
  // i + prefetch_lookahead at the ith iteration of the implemented loop.
  // keys and values must have the same size.
  virtual Status Lookup(KeyContext... key_context, ValueType values) const = 0;
};

// Interface for containers which support lookups with prefetching.
template <typename ValueType, typename... KeyContext>
class LookupWithPrefetchInterface : public virtual SynchronizedInterface {
 public:
  using value_type = ValueType;

  // Lookup the values for keys and store them in values.
  // prefetch_lookahead is used to prefetch the key at index
  // i + prefetch_lookahead at the ith iteration of the implemented loop.
  // keys and values must have the same size.
  virtual Status Lookup(KeyContext... key_context, ValueType values,
                        int64 prefetch_lookahead) const = 0;
};

// Interface for containers with size concepts.
// Implementations must guarantee thread-safety when GetMutex is used to
// synchronize method access.
class SizeInterface : public virtual SynchronizedInterface {
 public:
  // Returns the number of elements in the container.
  virtual uint64 Size() const = 0;
};

// Interface for tables which can be initialized from key and value arguments.
template <typename ValueType, typename... KeyContext>
class KeyValueTableInitializerInterface : public virtual SynchronizedInterface {
 public:
  using value_type = ValueType;

  // Lookup the values for keys and store them in values.
  // prefetch_lookahead is used to prefetch the key at index
  // i + prefetch_lookahead at the ith iteration of the implemented loop.
  // keys and values must have the same size.
  virtual Status Initialize(KeyContext... key_context, ValueType values) = 0;
};

}  // namespace tables
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LOOKUP_TABLES_RESOURCE_INTERFACE_TEMPLATES_H_
