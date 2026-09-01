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

#include "xla/hlo/ir/hlo_payload_deduplicator.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/backend_config.h"

namespace xla {
namespace {

TEST(HloPayloadDeduplicatorTest, DeduplicateString) {
  HloPayloadDeduplicator deduplicator;
  EXPECT_EQ(deduplicator.Deduplicate("config1"), 0);
  EXPECT_EQ(deduplicator.Deduplicate("config2"), 1);
  // Duplicate string should return the same ID.
  EXPECT_EQ(deduplicator.Deduplicate("config1"), 0);

  auto payloads = deduplicator.TakePayloads();
  EXPECT_EQ(payloads.size(), 2);
  EXPECT_EQ(payloads[0], "config1");
  EXPECT_EQ(payloads[1], "config2");
}

TEST(HloPayloadDeduplicatorTest, DeduplicatePointer) {
  HloPayloadDeduplicator deduplicator;
  auto wrapper0 = std::make_shared<BackendConfigWrapper>("config1");
  // Shares the same pointer (fast path).
  auto wrapper1 = wrapper0;
  // Different pointer but same string (fallback).
  auto wrapper2 = std::make_shared<BackendConfigWrapper>("config1");

  EXPECT_EQ(deduplicator.Deduplicate(wrapper0.get()), 0);
  EXPECT_EQ(deduplicator.Deduplicate(wrapper1.get()), 0);
  EXPECT_EQ(deduplicator.Deduplicate(wrapper2.get()), 0);

  auto payloads = deduplicator.TakePayloads();
  EXPECT_EQ(payloads.size(), 1);
  EXPECT_EQ(payloads[0], "config1");
}

TEST(HloPayloadDeduplicatorTest, DeduplicateWithBaseOffset) {
  HloPayloadDeduplicator deduplicator(5);
  EXPECT_EQ(deduplicator.Deduplicate("config1"), 5);
  EXPECT_EQ(deduplicator.Deduplicate("config2"), 6);
  EXPECT_EQ(deduplicator.Deduplicate("config1"), 5);

  auto payloads = deduplicator.TakePayloads();
  EXPECT_EQ(payloads.size(), 2);
  EXPECT_EQ(payloads[0], "config1");
  EXPECT_EQ(payloads[1], "config2");
}

TEST(HloPayloadDeduplicatorTest, TakePayloadsMovesMemory) {
  HloPayloadDeduplicator deduplicator;
  EXPECT_EQ(deduplicator.Deduplicate("config1"), 0);

  auto payloads1 = deduplicator.TakePayloads();
  EXPECT_EQ(payloads1.size(), 1);
  EXPECT_EQ(payloads1[0], "config1");

  // Subsequent take should be empty because it was moved.
  auto payloads2 = deduplicator.TakePayloads();
  EXPECT_EQ(payloads2.size(), 0);
}

}  // namespace
}  // namespace xla
