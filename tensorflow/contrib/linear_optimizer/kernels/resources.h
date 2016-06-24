/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/platform/fingerprint.h"
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

  // Platform independent, compact and unique (with very high probability)
  // representation of an example id. 'Ephemeral' because it shouldn't be put
  // in persistent storage, as its implementation may change in the future.
  //
  // The current probability of at least one collision for 1B example_ids is
  // approximately 10^-21 (ie 2^60 / 2^129).
  using EphemeralKey = Fprint128;

  // Makes a key for the supplied example_id, for compact storage.
  static EphemeralKey MakeKey(const string& example_id);

  struct Data {
    float dual = 0;
    float primal_loss = 0;
    float dual_loss = 0;
    float example_weight = 0;
  };

  // Accessor and mutator for the entry at Key. Accessor creates an entry with
  // default value (default constructed object) if the key is not present and
  // returns it.
  Data Get(const EphemeralKey& key) LOCKS_EXCLUDED(mu_);
  void Set(const EphemeralKey& key, const Data& data) LOCKS_EXCLUDED(mu_);

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
  // Backing container.
  //
  // sizeof(EntryPayload) =
  // sizeof(Key) + sizeof(Data) =
  // 16 + 16 = 32.
  //
  // So on average we use ~51.5 (32 + 19.5) bytes per entry in this table.
  using EphemeralKeyHasher = Fprint128Hasher;
  using DataByKey = std::unordered_map<EphemeralKey, Data, EphemeralKeyHasher>;

  // TODO(sibyl-Mooth6ku): Benchmark and/or optimize this.
  static const size_t kVisitChunkSize = 100;

  const string container_;
  const string solver_uuid_;

  // TODO(sibyl-Mooth6ku): Come up with a more efficient locking scheme.
  mutable mutex mu_;
  DataByKey data_by_key_ GUARDED_BY(mu_);

  friend class DataByExampleTest;
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LINEAR_OPTIMIZER_KERNELS_RESOURCES_H_
