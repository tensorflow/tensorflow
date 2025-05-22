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
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/pjrt_ifrt/pjrt_layout.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

TEST(PjRtLayoutSerDesTest, PjRtLayoutRoundTrip) {
  auto layout = PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(
      xla::LayoutUtil::MakeDescendingLayout(1)));

  TF_ASSERT_OK_AND_ASSIGN(auto serialized,
                          Serialize(*layout, /*options=*/nullptr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto deserialized,
      Deserialize<PjRtLayout>(serialized, /*options=*/nullptr));

  const auto* out_layout = llvm::dyn_cast<PjRtLayout>(deserialized.get());
  ASSERT_NE(out_layout, nullptr);
  EXPECT_EQ(*out_layout->pjrt_layout(), *layout->pjrt_layout());
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
