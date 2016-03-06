/* Copyright 2016 Google Inc. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_LINEAR_OPTIMIZER_KERNELS_RESOURCES_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_LINEAR_OPTIMIZER_KERNELS_RESOURCES_H_

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Resource for storing per-example data across many sessions.
// This class is thread-safe.
class DataByExample : public ResourceBase {
 public:
  // The container and solver_uuid are only used for debugging purposes.
  DataByExample(const string& container, const string& solver_uuid);

  virtual ~DataByExample();

  using Key = std::pair<uint64, uint32>;

  // Makes a key for the supplied example_id, for compact storage.
  static Key MakeKey(const string& example_id);

  struct Data {
    // TODO(rohananil): Add extra data needed for duality gap computation here.
    float dual;
  };

  // Accessor and mutator for the entry at Key. Creates an entry with default
  // value (0) if the key is not present.
  Data& operator[](const Key& key);

  string DebugString() override;

 private:
  struct KeyHash {
    size_t operator()(const Key& key) const;
  };

  const string container_;
  const string solver_uuid_;

  // TODO(katsiapis): Come up with a more efficient locking scheme.
  mutex mu_;

  // Backing container.
  //
  // sizeof(EntryPayload) = sizeof(Key) + sizeof(Data) = 16.
  // So on average we use ~35 bytes per entry in this table.
  std::unordered_map<Key, Data, KeyHash> duals_by_key;  // Guarded by mu_.
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LINEAR_OPTIMIZER_KERNELS_RESOURCES_H_
