/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/libraries/cub/cub_sort_types.h"

#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/primitive_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::AnyOf;

std::vector<PrimitiveType> AllValidPrimitiveTypes() {
  std::vector<PrimitiveType> types;
  for (int t = PrimitiveType_MIN; t <= PrimitiveType_MAX; ++t) {
    if (PrimitiveType_IsValid(t)) {
      types.push_back(static_cast<PrimitiveType>(t));
    }
  }
  return types;
}

TEST(CubSortTypesTest, PredicateAndValidatorAgreeAcrossAllPrimitiveTypes) {
  const std::vector<PrimitiveType> all_types = AllValidPrimitiveTypes();
  ASSERT_FALSE(all_types.empty());

  for (PrimitiveType key_type : all_types) {
    const bool keys_only_supported = IsCubSortSupported(key_type, std::nullopt);
    const absl::Status keys_only_status =
        ValidateCubSortTypes(key_type, std::nullopt);
    if (keys_only_supported) {
      EXPECT_THAT(keys_only_status, IsOk()) << PrimitiveType_Name(key_type);
    } else {
      EXPECT_THAT(keys_only_status,
                  StatusIs(absl::StatusCode::kInvalidArgument))
          << PrimitiveType_Name(key_type);
    }

    for (PrimitiveType val_type : all_types) {
      const bool pair_supported = IsCubSortSupported(key_type, val_type);
      const absl::Status pair_status = ValidateCubSortTypes(key_type, val_type);
      if (pair_supported) {
        EXPECT_THAT(pair_status, IsOk()) << PrimitiveType_Name(key_type) << ", "
                                         << PrimitiveType_Name(val_type);
      } else {
        EXPECT_THAT(pair_status, StatusIs(absl::StatusCode::kInvalidArgument))
            << PrimitiveType_Name(key_type) << ", "
            << PrimitiveType_Name(val_type);
      }
    }
  }
}

TEST(CubSortTypesTest, StructuralInvariantsOfSupportedTypes) {
  const std::vector<PrimitiveType> all_types = AllValidPrimitiveTypes();

  std::vector<PrimitiveType> supported_keys_only;
  std::vector<PrimitiveType> supported_pair_keys;

  for (PrimitiveType key_type : all_types) {
    if (IsCubSortSupported(key_type, std::nullopt)) {
      supported_keys_only.push_back(key_type);
      EXPECT_TRUE(primitive_util::IsArrayType(key_type))
          << PrimitiveType_Name(key_type);
      EXPECT_FALSE(primitive_util::IsComplexType(key_type))
          << PrimitiveType_Name(key_type);
      EXPECT_NE(key_type, PRED);
      if (primitive_util::IsArrayType(key_type)) {
        EXPECT_THAT(primitive_util::BitWidth(key_type), AnyOf(8, 16, 32, 64))
            << PrimitiveType_Name(key_type);
      }
    }

    bool any_val_supported = false;
    absl::flat_hash_map<int, bool> support_by_val_bitwidth;
    for (PrimitiveType val_type : all_types) {
      const bool supported = IsCubSortSupported(key_type, val_type);
      if (supported) {
        any_val_supported = true;
        // Pair keys must be a subset of keys-only keys (Onesweep pair shards
        // reuse the keys-only histogram kernel).
        EXPECT_TRUE(IsCubSortSupported(key_type, std::nullopt))
            << PrimitiveType_Name(key_type);
        EXPECT_TRUE(primitive_util::IsArrayType(val_type))
            << PrimitiveType_Name(val_type);
      }
      if (primitive_util::IsArrayType(val_type)) {
        // Pair value support depends only on bit width.
        const int bits = primitive_util::BitWidth(val_type);
        auto [it, inserted] = support_by_val_bitwidth.emplace(bits, supported);
        if (!inserted) {
          EXPECT_EQ(it->second, supported)
              << "key=" << PrimitiveType_Name(key_type)
              << " val=" << PrimitiveType_Name(val_type) << " bits=" << bits;
        }
      }
    }
    if (any_val_supported) {
      supported_pair_keys.push_back(key_type);
    }
  }

  // Non-emptiness guard so a trivially-false implementation cannot pass.
  EXPECT_FALSE(supported_keys_only.empty());
  ASSERT_FALSE(supported_pair_keys.empty());

  // Supported pairs factorize as SupportedPairKeys x SupportedPairValues.
  const PrimitiveType reference_key = supported_pair_keys.front();
  for (PrimitiveType pair_key : supported_pair_keys) {
    for (PrimitiveType val_type : all_types) {
      EXPECT_EQ(IsCubSortSupported(pair_key, val_type),
                IsCubSortSupported(reference_key, val_type))
          << "pair_key=" << PrimitiveType_Name(pair_key)
          << " reference_key=" << PrimitiveType_Name(reference_key)
          << " val=" << PrimitiveType_Name(val_type);
    }
  }
}

}  // namespace
}  // namespace xla::gpu
