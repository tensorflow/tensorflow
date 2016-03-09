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
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Resource for storing per-example data across many sessions. The data is
// operated on in a modify or append fashion (data can be modified or added, but
// never deleted).
//
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
    float dual = 0;
    float primal_loss = 0;
    float dual_loss = 0;
    float example_weight = 0;

    // Comparison operators for ease of testing.
    bool operator==(const Data& other) const { return dual == other.dual; }
    bool operator!=(const Data& other) const { return !(*this == other); }
  };

  // Accessor and mutator for the entry at Key. Accessor creates an entry with
  // default value (default constructed object) if the key is not present and
  // returns it.
  Data Get(const Key& key) LOCKS_EXCLUDED(mu_);
  void Set(const Key& key, const Data& data) LOCKS_EXCLUDED(mu_);

  // Visits all elements in this resource. The view of each element (Data) is
  // atomic, but the entirety of the visit is not (ie the visitor might see
  // different versions of the Data across elements).
  //
  // Returns OK on success or UNAVAILABLE if the number of elements in this
  // container has changed since the beginning of the visit (in which case the
  // visit cannot be completed and is aborted early, and computation can be
  // restarted).
  Status Visit(std::function<void(const Data& data)> visitor) const
      LOCKS_EXCLUDED(mu_);

  string DebugString() override;

 private:
  struct KeyHash {
    size_t operator()(const Key& key) const;
  };

  // Backing container.
  //
  // sizeof(EntryPayload) =
  // sizeof(Key) + sizeof(Data) =
  // 12 + 16 = 28.
  //
  // So on average we use ~47.5 (28 + 19.5) bytes per entry in this table.
  using DataByKey = std::unordered_map<Key, Data, KeyHash>;

  // TODO(katsiapis): Benchmark and/or optimize this.
  static const size_t kVisitChunkSize = 100;

  const string container_;
  const string solver_uuid_;

  // TODO(katsiapis): Come up with a more efficient locking scheme.
  mutable mutex mu_;
  DataByKey data_by_key_ GUARDED_BY(mu_);

  friend class DataByExampleTest;
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LINEAR_OPTIMIZER_KERNELS_RESOURCES_H_
