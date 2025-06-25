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
#include <utility>

#include <gtest/gtest.h>
#include "llvm/Support/Casting.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/pjrt_ifrt/pjrt_layout.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

class PjRtLayoutSerDesTest : public testing::TestWithParam<SerDesVersion> {
 public:
  PjRtLayoutSerDesTest() : version_(GetParam()) {}

  SerDesVersion version() const { return version_; }

 private:
  SerDesVersion version_;
};

TEST_P(PjRtLayoutSerDesTest, PjRtLayoutRoundTrip) {
  auto layout = PjRtLayout::Create(std::make_unique<xla::PjRtLayout>(
      xla::LayoutUtil::MakeDescendingLayout(1)));

  auto options = std::make_unique<SerializeOptions>(version());
  TF_ASSERT_OK_AND_ASSIGN(auto serialized,
                          Serialize(*layout, std::move(options)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto deserialized,
      Deserialize<PjRtLayout>(serialized, /*options=*/nullptr));

  const auto* out_layout = llvm::dyn_cast<PjRtLayout>(deserialized.get());
  ASSERT_NE(out_layout, nullptr);
  EXPECT_EQ(*out_layout->pjrt_layout(), *layout->pjrt_layout());
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion, PjRtLayoutSerDesTest,
    testing::ValuesIn(test_util::AllSupportedSerDesVersions()));

}  // namespace
}  // namespace ifrt
}  // namespace xla
