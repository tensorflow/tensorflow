/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_FINGERPRINT_LOOKUP_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_FINGERPRINT_LOOKUP_H_

#include <cstddef>
#include <deque>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/stringpiece.h"

namespace tensorflow {
namespace tpu {

// A class that holds the key-value pair of fingerprints. By calling the
// Register method, this class can map the key to the value. Note that this
// class holds invariant key-value pairs. That is, it does not allow updating
// key-value pairs, nor N-key-to-1-value and 1-key-to-M-value pairs. If such
// cases occur, the class keeps the earliest registered pairs and discards any
// violating pairs.
//
// Example:
//  TpuFingerprintLookup fingerprint_lookup;
//
//  // Register key-intermediate pair.
//  fingerprint_lookup.RegisterKeyValuePair("key1", "intermediate1");
//  // Register intermediate-value pair.
//  fingerprint_lookup.RegisterKeyValuePair("intermediate1", "value1");
//
//  // Lookup fingerprint with key.
//  std::string fingerprint = fingerprint_lookup.Lookup("key1");
//
// TODO(chiachenc): use templates and add Unregister methods.
class TpuFingerprintLookup : public ResourceBase {
 public:
  // Creates an instance of TpuFingerprintLookup.
  static TpuFingerprintLookup* Create();

  // Register key-intermediate pair
  void RegisterKeyAndIntermediatePair(uint64 key, uint64 intermediate);

  // Register intermediate-value pair. A successful registration requires a
  // preceding RegisterKeyAndIntermediatePair. Return true if successfully
  // registering a key-value pair; otherwise, return false.
  bool RegisterIntermediateAndValuePair(uint64 intermediate, std::string value);

  // Look up fingerprint with key.
  // Return std::nullopt if not found.
  std::optional<::tensorflow::StringPiece> Lookup(uint64 key);

  size_t num_valid() {
    absl::MutexLock lock(&mu_);
    return key_to_value_.size();
  }

  std::string DebugString() const override { return "TpuFingerprintLookup"; }

 private:
  explicit TpuFingerprintLookup() {}

  absl::Mutex mu_;
  // Main storage for lookup
  absl::node_hash_map<uint64, std::string> key_to_value_ ABSL_GUARDED_BY(mu_);

  // An auxiliary storage to ensure 1-to-1 and invariant key-value pair
  absl::node_hash_map<std::string, uint64> value_to_key_ ABSL_GUARDED_BY(mu_);

  // An auxiliary storage to keep intermediate-key pairs.
  absl::flat_hash_map<uint64, uint64> intermediate_to_key_ ABSL_GUARDED_BY(mu_);

  TpuFingerprintLookup(const TpuFingerprintLookup&) = delete;
  TpuFingerprintLookup& operator=(const TpuFingerprintLookup&) = delete;
};
}  // namespace tpu
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_FINGERPRINT_LOOKUP_H_
