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

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

TEST(PjRtExecutableImplTest, HloOutputCallbackWrapper) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  bool invoked = false;
  auto hlo_cb = std::make_unique<xla::HloOutputCallback>();
  hlo_cb->callback_id = 42;
  hlo_cb->num_operands = 1;
  hlo_cb->callback =
      [&invoked](
          int64_t replica_id, int64_t partition_id,
          absl::Span<std::shared_ptr<const xla::Literal> const> literals) {
        invoked = true;
        EXPECT_EQ(replica_id, 0);
        EXPECT_EQ(partition_id, 0);
        ASSERT_EQ(literals.size(), 1);
        ASSERT_NE(literals[0], nullptr);
        EXPECT_EQ(literals[0]->Get<int32_t>({}), 123);
      };

  auto loaded_host_callback = tsl::MakeRef<PjRtHloOutputLoadedHostCallback>(
      client.get(), std::move(hlo_cb));

  EXPECT_EQ(loaded_host_callback->hlo_output_callback().callback_id, 42);
  EXPECT_EQ(loaded_host_callback->hlo_output_callback().num_operands, 1);

  ASSERT_OK_AND_ASSIGN(
      auto lit, xla::Literal::Make(xla::ShapeUtil::MakeShape(xla::S32, {})));
  lit.Set<int32_t>({}, 123);
  auto shared_lit = std::make_shared<const xla::Literal>(std::move(lit));
  loaded_host_callback->hlo_output_callback().callback(
      0, 0, absl::MakeSpan(&shared_lit, 1));
  EXPECT_TRUE(invoked);
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
