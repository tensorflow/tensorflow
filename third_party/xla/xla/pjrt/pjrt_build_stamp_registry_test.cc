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

#include "xla/pjrt/pjrt_build_stamp_registry.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_build_stamp.h"
#include "xla/pjrt/pjrt_compiler_variant.h"
#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;

TEST(PjRtBuildStampRegistryTest, RegisterAndLookupCompiler) {
  PjRtBuildStampRegistry registry;
  int factory_calls = 0;
  auto factory = [&]() -> absl::StatusOr<PjRtBuildStamp> {
    factory_calls++;
    PjRtBuildStampProto proto;
    proto.set_source_hash("hash1");
    return PjRtBuildStamp::Create(std::move(proto));
  };

  ASSERT_OK(
      registry.RegisterBuildStampFactory("cuda", "custom_variant", factory));

  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull stamp,
                       registry.GetBuildStamp("cuda", "custom_variant"));
  EXPECT_EQ(stamp->source_hash(), "hash1");
  EXPECT_EQ(factory_calls, 1);

  // Subsequent lookups must be memoized without re-invoking the factory.
  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull stamp2,
                       registry.GetBuildStamp("cuda", "custom_variant"));
  EXPECT_EQ(stamp2, stamp);
  EXPECT_EQ(stamp2->source_hash(), "hash1");
  EXPECT_EQ(factory_calls, 1);

  // Subsequent lookup with convenience function must also be memoized.
  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull stamp3,
                       registry.GetBuildStamp("cuda", "custom_variant"));
  EXPECT_EQ(stamp3, stamp2);
  EXPECT_EQ(stamp3->source_hash(), "hash1");
  EXPECT_EQ(factory_calls, 1);
}

TEST(PjRtBuildStampRegistryTest, PjRtRegisterAndLookupCompiler) {
  int factory_calls = 0;
  auto factory = [&]() -> absl::StatusOr<PjRtBuildStamp> {
    factory_calls++;
    PjRtBuildStampProto proto;
    proto.set_source_hash("hash2");
    return PjRtBuildStamp::Create(std::move(proto));
  };

  PjRtRegisterBuildStampFactory("tpu", "linked", factory);
  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull stamp,
                       PjRtGetBuildStamp("tpu", "linked"));
  EXPECT_EQ(stamp->source_hash(), "hash2");
  EXPECT_EQ(factory_calls, 1);

  // Subsequent lookups must be memoized without re-invoking the factory.
  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull stamp2,
                       PjRtGetBuildStamp("tpu", "linked"));
  EXPECT_EQ(stamp2, stamp);
  EXPECT_EQ(stamp2->source_hash(), "hash2");
  EXPECT_EQ(factory_calls, 1);

  // Subsequent lookup with default variant must also be memoized.
  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull stamp3,
                       PjRtGetBuildStamp("tpu"));
  EXPECT_EQ(stamp3, stamp2);
  EXPECT_EQ(stamp3->source_hash(), "hash2");
  EXPECT_EQ(factory_calls, 1);

  // Subsequent lookup with global registry must also be memoized.
  ASSERT_OK_AND_ASSIGN(
      const PjRtBuildStamp* absl_nonnull stamp4,
      PjRtBuildStampRegistry::Global().GetBuildStamp("tpu", "linked"));
  EXPECT_EQ(stamp4, stamp2);
  EXPECT_EQ(stamp4->source_hash(), "hash2");
  EXPECT_EQ(factory_calls, 1);
}

TEST(PjRtBuildStampRegistryTest, RegisterAndLookupDefaultVariant) {
  PjRtBuildStampRegistry registry;
  auto linked_factory = []() -> absl::StatusOr<PjRtBuildStamp> {
    PjRtBuildStampProto proto;
    proto.set_source_hash("linked_hash");
    return PjRtBuildStamp::Create(std::move(proto));
  };

  ASSERT_OK(registry.RegisterBuildStampFactory("tpu", linked_factory));
  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull default_stamp,
                       registry.GetBuildStamp("tpu"));
  EXPECT_EQ(default_stamp->source_hash(), "linked_hash");

  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull linked_stamp,
                       registry.GetBuildStamp("tpu", kLinkedVariant));
  EXPECT_EQ(linked_stamp, default_stamp);
  EXPECT_EQ(linked_stamp->source_hash(), "linked_hash");
}

TEST(PjRtBuildStampRegistryTest, RegisterAndLookupMultipleVariants) {
  PjRtBuildStampRegistry registry;
  auto tpu_linked_factory = []() -> absl::StatusOr<PjRtBuildStamp> {
    PjRtBuildStampProto proto;
    proto.set_source_hash("tpu_linked");
    return PjRtBuildStamp::Create(std::move(proto));
  };
  auto tpu_forge_factory = []() -> absl::StatusOr<PjRtBuildStamp> {
    PjRtBuildStampProto proto;
    proto.set_source_hash("tpu_forge");
    return PjRtBuildStamp::Create(std::move(proto));
  };

  ASSERT_OK(registry.RegisterBuildStampFactory("tpu", kLinkedVariant,
                                               tpu_linked_factory));
  ASSERT_OK(
      registry.RegisterBuildStampFactory("tpu", "forge", tpu_forge_factory));

  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull tpu_linked_stamp,
                       registry.GetBuildStamp("tpu", kLinkedVariant));
  EXPECT_EQ(tpu_linked_stamp->source_hash(), "tpu_linked");

  ASSERT_OK_AND_ASSIGN(const PjRtBuildStamp* absl_nonnull tpu_forge_stamp,
                       registry.GetBuildStamp("tpu", "forge"));
  EXPECT_EQ(tpu_forge_stamp->source_hash(), "tpu_forge");
}

TEST(PjRtBuildStampRegistryTest, DuplicateRegistrationFails) {
  PjRtBuildStampRegistry registry;
  auto factory = []() -> absl::StatusOr<PjRtBuildStamp> {
    PjRtBuildStampProto proto;
    proto.set_source_hash("h");
    return PjRtBuildStamp::Create(std::move(proto));
  };

  ASSERT_OK(registry.RegisterBuildStampFactory("tpu", "variant_a", factory));
  EXPECT_THAT(registry.RegisterBuildStampFactory("tpu", "variant_a", factory),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(PjRtBuildStampRegistryTest, LookupNonExistentFails) {
  PjRtBuildStampRegistry registry;
  EXPECT_THAT(registry.GetBuildStamp("tpu", "unknown_variant"),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace xla
