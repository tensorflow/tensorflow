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

#include "xla/backends/cpu/runtime/resource_use.h"

#include <memory>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/types/span.h"

namespace xla::cpu {

std::shared_ptr<Resource> Resource::Create(Kind kind) {
  return absl::WrapUnique(new Resource(kind));
}

Resource::Resource(Kind kind) : kind_(kind) {}

ResourceUse::ResourceUse(std::shared_ptr<Resource> resource,
                         ResourceAccess access)
    : resource_(resource), access_(access) {}

ResourceUse::ReadWriteSet::ReadWriteSet() = default;

void ResourceUse::ReadWriteSet::Add(ResourceUse use) {
  switch (use.access()) {
    case ResourceUse::kRead:
      read_.insert(use.resource());
      break;
    case ResourceUse::kWrite:
      write_.insert(use.resource());
      break;
  }
}

void ResourceUse::ReadWriteSet::AddAll(absl::Span<const ResourceUse> uses) {
  for (const auto& use : uses) Add(use);
}

bool ResourceUse::ReadWriteSet::HasConflicts(const ResourceUse& use) const {
  return use.access() == ResourceAccess::kWrite
             ? write_.contains(use.resource()) || read_.contains(use.resource())
             : write_.contains(use.resource());
}

bool ResourceUse::ReadWriteSet::HasConflicts(
    absl::Span<const ResourceUse> uses) const {
  return absl::c_any_of(
      uses, [&](const ResourceUse& use) { return HasConflicts(use); });
}

bool ResourceUse::ReadWriteSet::HasConflicts(const ReadWriteSet& other) {
  return absl::c_any_of(other.read_,
                        [&](const std::shared_ptr<Resource>& resource) {
                          return HasConflicts(ResourceUse::Read(resource));
                        }) ||
         absl::c_any_of(other.write_,
                        [&](const std::shared_ptr<Resource>& resource) {
                          return HasConflicts(ResourceUse::Write(resource));
                        });
}

}  // namespace xla::cpu
