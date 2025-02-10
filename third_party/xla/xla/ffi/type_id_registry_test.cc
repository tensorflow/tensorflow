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

#include "xla/ffi/type_id_registry.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::ffi {
namespace {

using ::testing::HasSubstr;

TEST(TypeIdRegistryTest, RegisterExternalTypeId) {
  TF_ASSERT_OK_AND_ASSIGN(auto type_id,
                          TypeIdRegistry::RegisterExternalTypeId("foo"));
  EXPECT_GE(type_id.value(), 0);

  auto duplicate_type_id = TypeIdRegistry::RegisterExternalTypeId("foo");
  EXPECT_THAT(duplicate_type_id.status().message(),
              HasSubstr("already registered for type name foo"));
}

TEST(TypeIdRegistryTest, RegisterInternalTypeId) {
  auto int32_type_id = TypeIdRegistry::GetTypeId<int32_t>();
  auto int64_type_id = TypeIdRegistry::GetTypeId<int64_t>();
  EXPECT_NE(int32_type_id, int64_type_id);
}

}  // namespace
}  // namespace xla::ffi
