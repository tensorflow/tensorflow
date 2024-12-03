/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_RESOURCE_USE_H_
#define XLA_BACKENDS_CPU_RUNTIME_RESOURCE_USE_H_

#include <memory>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"

namespace xla::cpu {

// `Resource` models a run time resource that imposes ordering on the thunk
// execution in addition to thunk buffer uses.
class Resource {
 public:
  enum class Kind {
    // Side-effecting operations (i.e., infeed and outfeed) define their
    // execution order via token dependencies. We rely on token resource to
    // enforce ordering at run time.
    kToken,

    // Collective operations must be executed in the same order as they are
    // defined in the HLO module. We rely on collective communicator resource
    // to enforce ordering at run time.
    kCollectiveCommunicator
  };

  static constexpr Kind kToken = Kind::kToken;
  static constexpr Kind kCollectiveCommunicator = Kind::kCollectiveCommunicator;

  static std::shared_ptr<Resource> Create(Kind kind);

  Kind kind() const { return kind_; }

 private:
  explicit Resource(Kind kind);
  Kind kind_;
};

// For consistency with BufferUse, we model resource uses as writes or reads
// to and from resource. Resources have referential equality: we rely on
// comparing pointers to check if resource is the same or not.
class ResourceUse {
 public:
  enum class ResourceAccess { kRead, kWrite };

  static constexpr ResourceAccess kRead = ResourceAccess::kRead;
  static constexpr ResourceAccess kWrite = ResourceAccess::kWrite;

  static ResourceUse Read(std::shared_ptr<Resource> resource) {
    return ResourceUse(std::move(resource), ResourceAccess::kRead);
  }

  static ResourceUse Write(std::shared_ptr<Resource> resource) {
    return ResourceUse(std::move(resource), ResourceAccess::kWrite);
  }

  const std::shared_ptr<Resource>& resource() const { return resource_; }
  ResourceAccess access() const { return access_; }

  // ReadWriteSet tracks a set of read and write resources.
  class ReadWriteSet {
   public:
    ReadWriteSet();

    void Add(ResourceUse use);
    void AddAll(absl::Span<const ResourceUse> uses);

    // Returns true if any of the resource use(s) has a conflict with tracked
    // resource reads or writes.
    bool HasConflicts(const ResourceUse& use) const;
    bool HasConflicts(absl::Span<const ResourceUse> uses) const;
    bool HasConflicts(const ReadWriteSet& other);

   private:
    absl::flat_hash_set<std::shared_ptr<Resource>> read_;
    absl::flat_hash_set<std::shared_ptr<Resource>> write_;
  };

  bool operator==(const ResourceUse& other) const {
    return resource_ == other.resource_ && access_ == other.access_;
  }

  bool operator!=(const ResourceUse& other) const { return !(*this == other); }

  template <typename H>
  friend H AbslHashValue(H h, const ResourceUse& use) {
    return H::combine(std::move(h), use.resource_, use.access_);
  }

 private:
  ResourceUse(std::shared_ptr<Resource> resource, ResourceAccess access);
  std::shared_ptr<Resource> resource_;
  ResourceAccess access_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_RESOURCE_USE_H_
