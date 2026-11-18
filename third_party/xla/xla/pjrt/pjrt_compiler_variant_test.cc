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

#include "xla/pjrt/pjrt_compiler_variant.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_common.h"
#include "tsl/platform/fingerprint.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

constexpr absl::string_view kTestPlatform = "variant_picker_test_platform";
const PjRtPlatformId kTestPlatformId = tsl::Fingerprint64(kTestPlatform);

TEST(CompilerVariantFromNameTest, MapEmptyAndLinkedToLinkedVariantId) {
  EXPECT_EQ(CompilerVariantFromName(""), LinkedCompilerVariantId());
  EXPECT_EQ(CompilerVariantFromName(kLinkedVariant), LinkedCompilerVariantId());
  EXPECT_NE(CompilerVariantFromName("remote"), LinkedCompilerVariantId());
}

TEST(PjRtCompilerVariantPickerRegistryTest, DefaultsToLinked) {
  PjRtCompilerVariantPickerRegistry registry;
  EXPECT_THAT(registry.PickVariantName(kTestPlatformId),
              IsOkAndHolds(kLinkedVariant));
  EXPECT_THAT(registry.PickVariantName(kTestPlatform),
              IsOkAndHolds(kLinkedVariant));
  EXPECT_THAT(registry.PickVariant(kTestPlatformId),
              IsOkAndHolds(LinkedCompilerVariantId()));
}

TEST(PjRtCompilerVariantPickerRegistryTest,
     HasSamePickerForPlatformNameAndCorrespondingPlatformId) {
  PjRtCompilerVariantPickerRegistry registry;
  PjRtRegisterCompilerVariantPicker(
      kTestPlatform,
      []() -> absl::StatusOr<std::string> { return "variant_a"; },
      /*is_weak=*/false, registry);

  EXPECT_THAT(registry.PickVariantName(kTestPlatformId),
              IsOkAndHolds("variant_a"));
  EXPECT_THAT(registry.PickVariantName(kTestPlatform),
              IsOkAndHolds("variant_a"));
  EXPECT_THAT(registry.PickVariant(kTestPlatformId),
              IsOkAndHolds(CompilerVariantFromName("variant_a")));
  EXPECT_THAT(registry.PickVariant(kTestPlatform),
              IsOkAndHolds(CompilerVariantFromName("variant_a")));
}

TEST(PjRtCompilerVariantPickerRegistryTest,
     NormalizesEmptyVariantNameToLinkedVariantId) {
  PjRtCompilerVariantPickerRegistry registry;
  registry.RegisterVariantPicker(
      kTestPlatformId, []() -> absl::StatusOr<std::string> { return ""; });

  EXPECT_THAT(registry.PickVariantName(kTestPlatformId),
              IsOkAndHolds(kLinkedVariant));
  EXPECT_THAT(registry.PickVariant(kTestPlatformId),
              IsOkAndHolds(LinkedCompilerVariantId()));
}

TEST(PjRtCompilerVariantPickerRegistryTest, StrongOverridesWeakInEitherOrder) {
  {
    PjRtCompilerVariantPickerRegistry registry;
    registry.RegisterVariantPicker(
        kTestPlatform, []() -> absl::StatusOr<std::string> { return "weak_v"; },
        /*is_weak=*/true);
    registry.RegisterVariantPicker(
        kTestPlatform,
        []() -> absl::StatusOr<std::string> { return "strong_v"; },
        /*is_weak=*/false);
    EXPECT_THAT(registry.PickVariant(kTestPlatformId),
                IsOkAndHolds(CompilerVariantFromName("strong_v")));
  }

  {
    PjRtCompilerVariantPickerRegistry registry;
    registry.RegisterVariantPicker(
        kTestPlatform,
        []() -> absl::StatusOr<std::string> { return "strong_v"; },
        /*is_weak=*/false);
    registry.RegisterVariantPicker(
        kTestPlatform, []() -> absl::StatusOr<std::string> { return "weak_v"; },
        /*is_weak=*/true);
    EXPECT_THAT(registry.PickVariant(kTestPlatformId),
                IsOkAndHolds(CompilerVariantFromName("strong_v")));
  }
}

TEST(PjRtCompilerVariantPickerRegistryTest,
     PropagatesPickerErrorsWithoutCaching) {
  PjRtCompilerVariantPickerRegistry registry;
  int calls = 0;
  registry.RegisterVariantPicker(
      kTestPlatformId, [&calls]() -> absl::StatusOr<std::string> {
        if (++calls == 1) {
          return absl::UnavailableError("transient picker error");
        }
        return "variant_a";
      });

  EXPECT_THAT(registry.PickVariant(kTestPlatformId),
              StatusIs(absl::StatusCode::kUnavailable,
                       HasSubstr("transient picker error")));
  EXPECT_THAT(registry.PickVariant(kTestPlatformId),
              IsOkAndHolds(CompilerVariantFromName("variant_a")));
}

TEST(PjRtCompilerVariantPickerRegistryTest, FreeFunctionDefaultsToGlobal) {
  const absl::string_view kGlobalTestPlatform =
      "global_variant_picker_test_platform";
  PjRtRegisterCompilerVariantPicker(
      kGlobalTestPlatform,
      []() -> absl::StatusOr<std::string> { return "variant_b"; });

  EXPECT_THAT(PjRtCompilerVariantPickerRegistry::Global().PickVariant(
                  kGlobalTestPlatform),
              IsOkAndHolds(CompilerVariantFromName("variant_b")));

  const PjRtPlatformId global_id_platform =
      tsl::Fingerprint64("global_id_variant_picker_test_platform");
  PjRtRegisterCompilerVariantPicker(
      global_id_platform,
      []() -> absl::StatusOr<std::string> { return "variant_a"; });
  EXPECT_THAT(PjRtCompilerVariantPickerRegistry::Global().PickVariant(
                  global_id_platform),
              IsOkAndHolds(CompilerVariantFromName("variant_a")));
}

}  // namespace
}  // namespace xla
