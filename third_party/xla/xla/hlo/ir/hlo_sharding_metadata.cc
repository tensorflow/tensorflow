/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_sharding_metadata.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_domain_metadata.h"
#include "xla/hlo/ir/hlo_sharding.h"

namespace xla {

namespace {

bool ShardingMatches(const HloSharding& sharding1,
                     const HloSharding& sharding2) {
  auto single_sharding1 = sharding1.ExtractSingleSharding();
  if (single_sharding1) {
    auto single_sharding2 = sharding2.ExtractSingleSharding();
    if (single_sharding2) {
      return *single_sharding1 == single_sharding2;
    }
  }
  // Anything which is not unique across all elements, gets a full sharding
  // compare.
  return sharding1 == sharding2;
}

}  // namespace

std::unique_ptr<DomainMetadata> ShardingMetadata::Clone() const {
  std::unique_ptr<HloSharding> sharding;
  if (sharding_ != nullptr) {
    sharding = std::make_unique<HloSharding>(*sharding_);
  }
  return std::make_unique<ShardingMetadata>(std::move(sharding));
}

bool ShardingMetadata::Matches(const DomainMetadata& other) const {
  const ShardingMetadata* other_ptr =
      dynamic_cast<const ShardingMetadata*>(&other);
  if (other_ptr == nullptr) {
    // If other is not a ShardingMetadata, then it is clearly a no match.
    return false;
  }
  if (sharding_ == nullptr) {
    return other_ptr->sharding_ == nullptr;
  }
  return other_ptr->sharding_ != nullptr
             ? ShardingMatches(*sharding_, *other_ptr->sharding_)
             : false;
}

std::string ShardingMetadata::ToString() const {
  return sharding_ != nullptr ? sharding_->ToString() : "{}";
}

/*static*/ absl::StatusOr<const ShardingMetadata*>
ShardingMetadata::ToShardingMetadata(const DomainMetadata* metadata) {
  if (metadata->Kind() != ShardingMetadata::KindName()) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        "ShardingMetadata normalizer called with incorrect domain metadata");
  }
  return static_cast<const ShardingMetadata*>(metadata);
}

}  // namespace xla
