/* Copyright 2025 The OpenXLA Authors.

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

#include <memory>

#include <gtest/gtest.h>
#include "llvm/Support/Casting.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

TEST(LayoutSerDesTest, CompactLayoutRoundTrip) {
  TF_ASSERT_OK_AND_ASSIGN(auto layout, CompactLayout::Create({1, 0}));

  TF_ASSERT_OK_AND_ASSIGN(auto serialized,
                          Serialize(*layout, /*options=*/nullptr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto deserialized,
      Deserialize<CompactLayout>(serialized, /*options=*/nullptr));

  const auto* out_layout = llvm::dyn_cast<CompactLayout>(deserialized.get());
  ASSERT_NE(out_layout, nullptr);
  EXPECT_EQ(out_layout->major_to_minor(), layout->major_to_minor());
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
